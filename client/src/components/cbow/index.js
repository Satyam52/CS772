import React, { useEffect, useState } from "react";
import {
  MDBContainer,
  MDBRow,
  MDBCol,
  MDBCard,
  MDBCardHeader,
  MDBCardBody,
  MDBCardFooter,
  MDBIcon,
  MDBBtn,
  MDBTypography,
} from "mdb-react-ui-kit";
import { MDBTable, MDBTableHead, MDBTableBody } from 'mdb-react-ui-kit';
import axios from 'axios'


const Table = ({res}) => {
    const {input, output} = res;
    return( 
    <MDBTable>
    <MDBTableBody>
     {input && output && input.length>0 && output.length>0 &&
     <>
     <tr>
        {input.map(item=>(
            <td>{item}</td>
        ))}
      </tr>
      <tr>
        {output.map(item=>(
          <td>{item}</td>
        ))}
      </tr>
      </> }
    </MDBTableBody>
  </MDBTable>)
  }

export default function App() {
    const [word1, setWord1] = useState('')
    const [word2, setWord2] = useState('')
    const [word3, setWord3] = useState('')
    const [res, setRes] = useState([])

    const onSend = async()=>{
        try {
            const res = await axios.post('http://localhost:8000/api/v1/cbow',{'words':[word1, word2, word3]})
            setRes(res.data.words)
        } catch (error) {
            console.log(error)
        }
    }

  return (
    <>
    <MDBContainer fluid className="py-5" style={{ backgroundColor: "#eee" }}>
      <MDBRow className="d-flex justify-content-center">
        <MDBCol md="10" lg="8" xl="6">
          <MDBCard id="chat2" style={{ borderRadius: "15px" }}>
            <MDBCardHeader className="d-flex justify-content-between align-items-center p-3">
                <h5 className="mb-0">CBOW demo</h5>
              <MDBBtn onClick={()=>{
                setRes([])
                setWord1('');
                setWord2('');
                setWord3('');
              }} color="primary" size="sm" rippleColor="dark">
                Reset
              </MDBBtn>
            </MDBCardHeader>

            <MDBCardFooter className="text-muted d-flex justify-content-start align-items-center p-3 z-100">
              <input
                type="text"
                class="form-control form-control-lg"
                id="exampleFormControlInput1"
                placeholder="King"
                onChange={(e)=>setWord1(e.target.value)}
              ></input><b>:</b>
               <input
                type="text"
                class="form-control form-control-lg"
                id="exampleFormControlInput1"
                placeholder="Queen"
                onChange={(e)=>setWord2(e.target.value)}
              ></input><b>::</b>
               <input
                type="text"
                class="form-control form-control-lg"
                id="exampleFormControlInput1"
                placeholder="Boy"
                onChange={(e)=>setWord3(e.target.value)}
              ></input><b>:</b>
              <MDBBtn onClick={()=>onSend()}  className="ms-3" href="#!">
                <MDBIcon fas icon="arrow-right" size='2x' />
              </MDBBtn>
            </MDBCardFooter>
          </MDBCard>
          {res.length>0 && res.map(item=>(<div>{item}</div>))}
        </MDBCol>
      </MDBRow>


      </MDBContainer>
      </>
  );
}